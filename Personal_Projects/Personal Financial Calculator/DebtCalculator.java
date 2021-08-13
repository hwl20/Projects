package DebtPaymentCalculator;

import java.util.Scanner;

public class DebtCalculator{
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("NOTE: This system assumes fixed interest rate throughout tenure instead of a variable interest rate");
        System.out.println("*VERY IMPORTANT: Please do not include SYMBOLS in your inputs");
        System.out.print("Bank loaned from: ");
        String bank = scanner.next();
        System.out.println("\nNote that you are required to pay a processing fee amounting to $350 on average");
        System.out.print("Type in amount owed: ");
        double owed = scanner.nextDouble();

        System.out.println("Enter type of loan: \n\t1-Flat Annual Interest Rate(p.a)\n\t2-Effective Interest Rate(p.a)");
        System.out.print("Response [1/2]: ");
        int choice = scanner.nextInt();

        switch (choice) {
            case 1:
                System.out.print("\nType in intended duration of loan (years): ");
                double years = scanner.nextDouble();
                System.out.print("Enter Flat Annual Interest Rate(% p.a): ");
                double interest = scanner.nextDouble() / 100;
                double totalAmount = owed + owed * interest * years;
                double amountPerMonth = totalAmount / years / 12;
                System.out.format("Amount per month for loaning with %s is: $%.2f",bank,amountPerMonth);
                break;
            case 2:
                System.out.println("\nI will be providing:\n 1-Intended Monthly repayment Amount\n 2-Intended duration of loan (years)");
                System.out.print("Response [1/2]: ");
                int byMthAmt = scanner.nextInt();
                if (byMthAmt == 1) {
                    System.out.print("Amount intended to be paid per month: ");
                    double payment = scanner.nextDouble();
                    System.out.print("Enter Effective Interest Rate(% p.a): ");
                    double interestE = scanner.nextDouble() / 100 / 12;
                    double totalI = 1+interestE;
                    //System.out.println(totalI);
                    int months = 0;
                    while (owed > 0) {
                        owed = owed * (totalI) - payment;
                        //System.out.println(owed);
                        months++;
                    }

                    System.out.format("Duration for paying $%.2f to %s every month is: %d months", payment, bank, months);
                }
                else{
                    System.out.print("Type in intended duration of loan (years): ");
                    double dn = scanner.nextDouble()*12;
                    System.out.print("Enter Effective Interest Rate(% p.a): ");
                    double interestE = scanner.nextDouble() / 100 / 12;
                    double interestSum = 0;
                    for (int i=0; i< dn; i++){
                        interestSum += Math.pow((1+interestE),i);
                    }
                    //System.out.println(interestSum); //Just to check
                    double amountPerMth = owed*Math.pow((1+interestE),dn)/interestSum;
                    System.out.format("Amount per month for loaning with %s is: $%.2f",bank,amountPerMth);
                }
        }
    }
}