package DcaCalculator;

import java.util.Scanner;

public class dca2 {
    public static void tryout2() {
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
        System.out.format("Putting $%d per month for %.0f years and $%d per month for %.0f years will return: ", monthlyAmount, year, monthlyAmount2, year2);
        System.out.format("\nAmount at the end of %.0f years is: $%.2f", year+year2, amount);
    }
}
