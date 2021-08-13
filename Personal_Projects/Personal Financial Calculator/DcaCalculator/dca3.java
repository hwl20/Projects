package DcaCalculator;

import java.util.Scanner;

public class dca3 {
    public static void tryout3() {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Type in amount put in monthly initially: ");
        int monthlyAmount = scanner.nextInt();
        double yearlyAmount = monthlyAmount * 12;

        System.out.format("Time horizon for $%d: ", monthlyAmount);
        double year = scanner.nextInt();


        System.out.print("\nType in amount put in monthly next: ");
        int monthlyAmount2 = scanner.nextInt();
        double yearlyAmount2 = monthlyAmount2 * 12;

        System.out.format("Time horizon for $%d: ", monthlyAmount2);
        double year2 = scanner.nextInt();


        System.out.print("\nType in amount put in monthly next: ");
        int monthlyAmount3 = scanner.nextInt();
        double yearlyAmount3 = monthlyAmount3 * 12;

        System.out.format("Time horizon for $%d: ", monthlyAmount3);
        double year3 = scanner.nextInt();

        System.out.print("Interest per annum (%): ");
        double interest = scanner.nextDouble()/100;

        double amount = 0;
        for (int i = 1; i<=year; i++){
            amount = amount*(1+interest) + yearlyAmount;
        }
        for (int i = 1; i<=year2; i++){
            amount = amount*(1+interest) + yearlyAmount2;
            System.out.println(amount);
        }
        for (int i = 1; i<=year3; i++){
            amount = amount*(1+interest) + yearlyAmount3;
            System.out.println(amount);
        }
        System.out.format("\nAmount at the end of %.0f years is: $%.2f", year+year2+year3, amount);
    }
}
